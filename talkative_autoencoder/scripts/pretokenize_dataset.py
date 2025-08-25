"""Pre-tokenize datasets to eliminate tokenization bottleneck during activation dumping."""

import json
import logging
from pathlib import Path

import dotenv
import hydra
from datasets import Dataset, load_dataset, load_from_disk
from omegaconf import DictConfig

from transformers import AutoTokenizer

dotenv.load_dotenv()

log = logging.getLogger(__name__)


def find_input_column_and_format(dataset: Dataset) -> (str, str):
    """
    Heuristically find the input column and determine its format (text, chat, or chat_list).

    Returns a tuple of (column_name, format_type), where format_type is 'text',
    'chat' (list of dicts), 'chat_list' (alternating list of strings), or
    'thinking' for datasets with 'question' and 'answer' columns.
    """
    column_names = dataset.column_names

    # 0. Check for "thinking" format.
    if "question" in column_names and "answer" in column_names:
        log.info("Detected 'thinking' format with 'question' and 'answer' columns.")
        # It doesn't really have a single input column, but we'll handle it
        # in the tokenize function. We return 'question' as a placeholder.
        return "question", "thinking"

    # 0b. gpt-oss 4-column format (reasoning effort optional)
    if "user_content" in column_names and "assistant_content" in column_names:
        log.info("Detected 'gpt_oss_reasoning' format with 'user_content'/'assistant_*' columns.")
        return "user_content", "gpt_oss_reasoning"

    # 1. Check for chat formats first by inspecting a few rows
    for i in range(min(10, len(dataset))):
        row = dataset[i]
        for col in column_names:
            if isinstance(row[col], list) and row[col]:
                first_item = row[col][0]
                if isinstance(first_item, dict) and all(k in first_item for k in ["role", "content"]):
                    return col, "chat"
                if isinstance(first_item, str):
                    log.info(f"Detected 'chat_list' format in column '{col}'.")
                    return col, "chat_list"

    # 2. If not chat, look for standard text columns
    text_keywords = ["text", "content", "document", "instruction"]
    for col in column_names:
        if any(keyword in col.lower() for keyword in text_keywords):
            if len(dataset) > 0 and isinstance(dataset[0][col], str):
                return col, "text"

    # 3. As a fallback, find the first column that is of string type
    if len(dataset) > 0:
        for col in column_names:
            if isinstance(dataset[0][col], str):
                return col, "text"

    raise ValueError(
        f"Could not automatically find a text or chat column in dataset with columns: {dataset.column_names}"
    )


def do_pretokenize(cfg: DictConfig):
    """The logic of pre-tokenizing a dataset, designed to be callable from other scripts."""
    # Setup logging
    log = logging.getLogger(__name__)

    # Get config values
    pretokenize_cfg = cfg.pretokenize
    activation_dumper_cfg = cfg.activation_dumper

    # Get dataset info
    dataset_name = activation_dumper_cfg["hf_dataset_name"]
    tokenizer_name = cfg["orig_tokenizer_name"]
    seq_len = activation_dumper_cfg["seq_len"]

    # Get pre-tokenization parameters
    output_dir_cfg = pretokenize_cfg.get("output_dir")
    num_proc = pretokenize_cfg.get("num_proc", 4)
    batch_size = pretokenize_cfg.get("batch_size", 1000)
    force = pretokenize_cfg.get("force", False)
    name_of_sub_dataset = pretokenize_cfg.get("name_of_sub_dataset", None)
    # Output directory - now includes seq_len for versioning
    if output_dir_cfg:
        output_dir = Path(output_dir_cfg)
    else:
        output_dir = (
            Path("data/pretokenized")
            / f"{cfg.model_name.replace('/', '_')}_{dataset_name.replace('/', '_')}_seq{seq_len}"
        )

    # Check if already exists and force flag
    if output_dir.exists() and not force:
        log.info(f"Pretokenized data already exists at {output_dir}")
        log.info("Use pretokenize.force=true to re-tokenize")
        return

    log.info(f"Pre-tokenizing dataset: {dataset_name}")
    log.info(f"Using tokenizer: {tokenizer_name}")
    log.info(f"Sequence length: {seq_len}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Using {num_proc} processes")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset Loading and Splitting ---
    train_split_name = "train"
    val_split_name_from_cfg = activation_dumper_cfg.get("val_hf_split")

    datasets_to_process = {}

    # Decide how to get train and validation datasets
    if not val_split_name_from_cfg or val_split_name_from_cfg == train_split_name:
        # Case: No validation split specified, or it's the same as train.
        # Create a validation split from the training data.
        log.info(f"No separate validation split specified. Creating one from '{train_split_name}'.")

        try:
            if Path(dataset_name).exists():
                full_train_dataset = load_from_disk(dataset_name)
            else:
                full_train_dataset = load_dataset(
                    dataset_name,
                    name=name_of_sub_dataset,
                    split=train_split_name,
                    trust_remote_code=True,
                    num_proc=num_proc,
                )
        except Exception as e:
            log.error(f"Failed to load '{train_split_name}' for '{dataset_name}': {e}")
            raise

        validation_split_size = pretokenize_cfg.get("validation_split_size", 0.05)
        seed = pretokenize_cfg.get("seed", 42)

        if len(full_train_dataset) < 2:
            raise ValueError(
                f"Train dataset '{train_split_name}' has fewer than 2 samples ({len(full_train_dataset)}), cannot create a validation split."
            )

        log.info(f"Splitting '{train_split_name}' with test_size={validation_split_size} and seed={seed}.")

        split_dataset = full_train_dataset.train_test_split(test_size=validation_split_size, shuffle=True, seed=seed)

        datasets_to_process["train"] = split_dataset["train"]
        datasets_to_process["validation"] = split_dataset["test"]

        log.info(f"Created train split with {len(datasets_to_process['train'])} samples.")
        log.info(f"Created validation split with {len(datasets_to_process['validation'])} samples.")

    else:
        # Case: A separate validation split is specified. Load both.
        log.info(f"Using specified train ('{train_split_name}') and validation ('{val_split_name_from_cfg}') splits.")

        if Path(dataset_name).exists():
            raise ValueError(
                "When providing a local dataset path (saved via save_to_disk), do not specify a separate validation split."
            )

        try:
            train_dataset = load_dataset(
                dataset_name, split=train_split_name, trust_remote_code=True, num_proc=num_proc
            )
            datasets_to_process["train"] = train_dataset
            log.info(f"Loaded train split '{train_split_name}' with {len(train_dataset)} samples.")
        except Exception as e:
            log.error(f"Failed to load train split '{train_split_name}': {e}")
            raise

        try:
            val_dataset = load_dataset(
                dataset_name, split=val_split_name_from_cfg, trust_remote_code=True, num_proc=num_proc
            )
            datasets_to_process["validation"] = val_dataset
            log.info(f"Loaded validation split '{val_split_name_from_cfg}' with {len(val_dataset)} samples.")
        except Exception as e:
            log.error(f"Failed to load specified validation split '{val_split_name_from_cfg}': {e}")
            raise

    # --- Find Text Column and Tokenize ---
    # Find text column from the first available dataset, assuming it's consistent across splits.
    if not datasets_to_process:
        log.error("No datasets were loaded or created. Cannot proceed.")
        return

    sample_dataset = next(iter(datasets_to_process.values()))
    input_column, data_format = find_input_column_and_format(sample_dataset)
    log.info(f"Using input column: '{input_column}' with format: '{data_format}' for all splits.")

    if data_format == "thinking":
        # The user wants this format only for gpt-oss models
        if "gpt-oss" not in cfg.model_name:
            log.warning(
                f"Detected 'thinking' dataset format, but model name '{cfg.model_name}'"
                " does not contain 'gpt-oss'. The chat template might not support the 'thinking' field."
            )
            raise ValueError(
                f"Detected 'thinking' dataset format, but model name '{cfg.model_name}'"
                " does not contain 'gpt-oss'. The chat template might not support the 'thinking' field."
            )

        def tokenize_function(examples):
            chats = []
            # The user's format has 'question' and 'answer'
            for i in range(len(examples["question"])):
                question = examples["question"][i]
                answer = examples["answer"][i]

                thinking = None
                content = answer

                if "<think>" in answer and "</think>" in answer:
                    # Split answer into thinking and content
                    think_part, content_part = answer.split("</think>", 1)
                    thinking = think_part.replace("<think>", "").strip()
                    content = content_part.strip()

                chat_message = [{"role": "user", "content": question}]
                assistant_message = {"role": "assistant", "content": content}

                # Add the thinking part if it exists
                if thinking:
                    assistant_message["thinking"] = thinking

                chat_message.append(assistant_message)
                chats.append(chat_message)

            # `add_generation_prompt=False` is used for training
            formatted_texts = [
                tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats
            ]

            return tokenizer(
                formatted_texts,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_special_tokens_mask=False,
                return_attention_mask=True,
                add_special_tokens=False,
            )

    elif data_format == "gpt_oss_reasoning":
        if "gpt-oss" not in cfg.model_name:
            log.warning(
                f"Detected 'gpt_oss_reasoning' dataset format, but model name '{cfg.model_name}'"
                " does not contain 'gpt-oss'. The chat template might not support the 'thinking' field."
            )
            raise ValueError(
                f"Detected 'gpt_oss_reasoning' dataset format, but model name '{cfg.model_name}'"
                " does not contain 'gpt-oss'. The chat template might not support the 'thinking' field."
            )

        if not tokenizer.chat_template:
            log.warning("Tokenizer does not have a chat_template. A gpt-oss template is required.")
            raise ValueError(
                "Tokenizer does not have a chat_template. Please use a tokenizer that supports gpt-oss chat templates."
            )

        def tokenize_function(examples):
            users = examples.get("user_content")
            assistants = examples.get("assistant_content")
            thinkings = examples.get("assistant_thinking")
            efforts = examples.get("system_reasoning_effort")

            chats = []
            effort_vals = []
            n = len(users)
            for i in range(n):
                user_msg = {"role": "user", "content": users[i]}
                assistant_msg = {"role": "assistant", "content": assistants[i] if assistants else ""}
                if thinkings and thinkings[i]:
                    assistant_msg["thinking"] = thinkings[i]
                chats.append([user_msg, assistant_msg])

                val = None
                if efforts:
                    val = efforts[i]
                effort_vals.append(val or "medium")

            formatted_texts = [
                tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=False,
                    reasoning_effort=effort_vals[i],
                )
                for i, chat in enumerate(chats)
            ]

            return tokenizer(
                formatted_texts,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_special_tokens_mask=False,
                return_attention_mask=True,
                add_special_tokens=False,
            )

    elif data_format in ["chat", "chat_list"]:
        if not tokenizer.chat_template:
            log.warning("Tokenizer does not have a chat_template. Applying a default ChatML template.")
            raise ValueError(
                "Tokenizer does not have a chat_template. Please use a tokenizer that supports chat templates."
            )
            # A common chatML template
            # tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"

        def tokenize_function(examples):
            conversations = examples[input_column]

            # If data is in list-of-strings format, convert it to list-of-dicts
            if data_format == "chat_list":
                formatted_conversations = []
                for conv_list in conversations:
                    formatted_conv = []
                    for i, message in enumerate(conv_list):
                        role = "user" if i % 2 == 0 else "assistant"
                        formatted_conv.append({"role": role, "content": message})
                    formatted_conversations.append(formatted_conv)
                conversations = formatted_conversations

            # If model does not support 'thinking' natively, inline it into content
            if "gpt-oss" not in cfg.model_name:
                prepared_conversations = []
                for conv in conversations:
                    prepared_conv = []
                    for msg in conv:
                        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("thinking"):
                            thinking_text = msg.get("thinking") or ""
                            content_text = msg.get("content") or ""
                            merged_content = f"<think>{thinking_text}</think>\n{content_text}".strip()
                            new_msg = {k: v for k, v in msg.items() if k != "thinking"}
                            new_msg["content"] = merged_content
                            prepared_conv.append(new_msg)
                        else:
                            prepared_conv.append(msg)
                    prepared_conversations.append(prepared_conv)
                conversations = prepared_conversations

            # apply_chat_template turns a list of dicts into a single string
            formatted_texts = [
                tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                for conv in conversations
            ]
            return tokenizer(
                formatted_texts,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_special_tokens_mask=False,
                add_special_tokens=False,
                return_attention_mask=True,
            )

    else:  # data_format == 'text'

        def tokenize_function(examples):
            return tokenizer(
                examples[input_column],
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_special_tokens_mask=False,  # Save space
                return_attention_mask=True,
            )

    # Print an example of the tokenized output for inspection
    try:
        example = sample_dataset[0]
        if data_format == "thinking":
            # Build a batch with one example for the "thinking" format
            batch = {
                "question": [example["question"]],
                "answer": [example["answer"]],
            }
            example_tokenized = tokenize_function(batch)
        elif data_format == "chat_list":
            example_tokenized = tokenize_function({input_column: [example[input_column]]})
        elif data_format == "gpt_oss_reasoning":
            batch = {
                "user_content": [example["user_content"]],
                "assistant_content": [example["assistant_content"]],
            }
            if "assistant_thinking" in example:
                batch["assistant_thinking"] = [example["assistant_thinking"]]
            if "system_reasoning_effort" in example:
                batch["system_reasoning_effort"] = [example["system_reasoning_effort"]]
            example_tokenized = tokenize_function(batch)
        else:
            example_tokenized = tokenize_function({input_column: example[input_column]})
        log.info(f"Example tokenized output: {example_tokenized}")
        example_decoded = tokenizer.decode(example_tokenized["input_ids"][0], skip_special_tokens=False)
        log.info(f"Example decoded output: {example_decoded}")
    except Exception as e:
        log.error(f"Error printing example tokenized output: {e}")

    for split_name, dataset in datasets_to_process.items():
        log.info(f"\nProcessing split: {split_name} ({len(dataset)} samples)")

        # Apply tokenization
        log.info("Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )

        # Save tokenized data
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Saving to {split_output_dir}")
        tokenized_dataset.save_to_disk(str(split_output_dir))

        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "tokenizer_name": tokenizer_name,
            "seq_len": seq_len,
            "num_samples": len(tokenized_dataset),
            "split": split_name,
            "columns": tokenized_dataset.column_names,
        }

        with open(split_output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"✓ Completed {split_name}: {len(tokenized_dataset)} samples")

    log.info("\n✓ Pre-tokenization complete!")
    log.info(f"To use, update your config to load from: {output_dir}")
    return output_dir


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logging for standalone script execution
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    do_pretokenize(cfg)


if __name__ == "__main__":
    main()
